use serde_derive::Deserialize;
use serde_enum_str::Deserialize_enum_str;
use serde_json::json;

use hull_white::HullWhite;
use lambda_http::{run, service_fn, Body, Error, Request, RequestExt, Response};

use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64;

fn build_response(code: u16, body: &str) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(code)
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Credentials", "true")
        .body::<Body>(body.into())?)
}
fn construct_error(e_message: &str) -> String {
    json!({ "err": e_message }).to_string()
}

fn yield_curve(curr_rate: f64, a: f64, b: f64, sig: f64) -> impl Fn(f64) -> f64 {
    move |t| {
        let at = (1.0 - (-a * t).exp()) / a;
        let ct =
            (b - sig.powi(2) / (2.0 * a.powi(2))) * (at - t) - sig.powi(2) * at.powi(2) / (4.0 * a);
        at * curr_rate - ct
    }
}
fn forward_curve(curr_rate: f64, a: f64, b: f64, sig: f64) -> impl Fn(f64) -> f64 {
    move |t| {
        let tmp = (-a * t).exp();
        b + tmp * (curr_rate - b) - (sig.powi(2) / (2.0 * a.powi(2))) * (1.0 - tmp).powi(2)
    }
}

fn generate_vasicek(curr_rate: f64, a: f64, b: f64, sig: f64, t: f64) -> impl Fn(f64) -> f64 {
    let tmp = (-a * t).exp();
    let mu = b * (1.0 - tmp) + curr_rate * tmp;
    let vol = sig * ((1.0 - (-2.0 * a * t).exp()) / (2.0 * a)).sqrt();
    move |random_number| mu + vol * random_number
}

const NUM_SIMS: usize = 500;

#[derive(Deserialize_enum_str, Debug)]
#[serde(rename_all = "lowercase")]
enum Asset {
    BOND,
    EDF,
    BONDCALL,
    BONDPUT,
    CAPLET,
    SWAP,
    SWAPTION,
    AMERICANSWAPTION,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BaseParameters {
    t: f64,
    r0: f64,
    a: f64,
    b: f64,
    sigma: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BondParameters {
    maturity: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EDFParameters {
    maturity: f64,
    tenor: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BondOptionParameters {
    maturity: f64,
    underlying_maturity: f64,
    strike: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CapletParameters {
    maturity: f64,
    tenor: f64,
    strike: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwapParameters {
    maturity: f64,
    tenor: f64,
    swap_rate: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwaptionParameters {
    maturity: f64,
    tenor: f64,
    num_swap_payments: usize,
    swap_rate: f64,
}

fn bin(min: f64, max: f64, num_bins: f64, elements: &[f64]) -> HashMap<String, usize> {
    let mut bins = HashMap::new();
    let range = max - min;
    let bin_width = range / num_bins;
    for element in elements.iter() {
        let key = if element == &max {
            format!("{:.4}-{:.4}", max - bin_width, max)
        } else {
            let lower_index = ((element - min) / bin_width).floor();
            let lower_bound = lower_index * bin_width + min;
            let upper_bound = (lower_index + 1.0) * bin_width + min;
            format!("{:.4}-{:.4}", lower_bound, upper_bound)
        };
        if let Some(x) = bins.get_mut(&key) {
            *x += 1;
        } else {
            bins.insert(key, 1);
        }
    }
    bins
}

const ONE_THIRD: f64 = 1.0 / 3.0;
fn combine_and_bin(min: f64, max: f64, elements: &[f64]) -> HashMap<String, usize> {
    let num_bins = (2.0 * (elements.len() as f64).powf(ONE_THIRD)).floor();
    bin(min, max, num_bins, elements)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let func = service_fn(func);
    run(func).await
}

async fn func(event: Request) -> Result<Response<Body>, Error> {
    match market_faas(event) {
        Ok(res) => Ok(build_response(200, &json!(res).to_string())?),
        Err(e) => Ok(build_response(400, &construct_error(&e.to_string()))?),
    }
}

fn mc_results<T>(num_sims: usize, func_to_sim: T) -> Vec<f64>
where
    T: Fn(f64) -> f64 + std::marker::Sync,
{
    let normal = StandardNormal;
    (0..num_sims)
        .into_par_iter()
        .map(|_index| {
            let norm = thread_rng().sample(normal);
            func_to_sim(norm)
        })
        .collect()
}

fn min_v(results: &[f64]) -> f64 {
    results.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}
fn max_v(results: &[f64]) -> f64 {
    results.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

fn mc<T>(num_sims: usize, func_to_sim: T) -> HashMap<String, usize>
where
    T: Fn(f64) -> f64 + std::marker::Sync,
{
    let results = mc_results(num_sims, &func_to_sim);
    let min = min_v(&results);
    let max = max_v(&results);
    combine_and_bin(min, max, &results)
}

//simplistic, but good enough
fn transform_days_to_year(t: f64) -> f64 {
    t / 365.0
}

fn market_faas(event: Request) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    let asset: Asset = event
        .path_parameters()
        .first("asset")
        .unwrap_or("bond")
        .parse::<Asset>()?;

    let BaseParameters { t, a, r0, b, sigma } = serde_json::from_reader(event.body().as_ref())?;
    let t = transform_days_to_year(t);
    let yield_fn = yield_curve(r0, a, b, sigma);
    let forward_fn = forward_curve(r0, a, b, sigma);
    let hull_white = HullWhite::init(a, sigma, &yield_fn, &forward_fn);
    let simulation = generate_vasicek(r0, a, b, sigma, t);
    match asset {
        Asset::BOND => {
            let BondParameters { maturity } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.bond_price_t(r_t, t, maturity)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::EDF => {
            let EDFParameters { maturity, tenor } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.euro_dollar_future_t(r_t, t, maturity, tenor)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::BONDCALL => {
            let BondOptionParameters {
                maturity,
                underlying_maturity,
                strike,
            } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.bond_call_t(r_t, t, maturity, underlying_maturity, strike)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::BONDPUT => {
            let BondOptionParameters {
                maturity,
                underlying_maturity,
                strike,
            } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.bond_put_t(r_t, t, maturity, underlying_maturity, strike)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::CAPLET => {
            let CapletParameters {
                maturity,
                strike,
                tenor,
            } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.caplet_t(r_t, t, maturity, tenor, strike)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::SWAP => {
            let SwapParameters {
                maturity,
                swap_rate,
                tenor,
            } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.swap_price_t(r_t, t, maturity, tenor, swap_rate)
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::SWAPTION => {
            let SwaptionParameters {
                maturity,
                swap_rate,
                num_swap_payments,
                tenor,
            } = serde_json::from_reader(event.body().as_ref())?;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white
                    .european_payer_swaption_t(
                        r_t,
                        t,
                        maturity,
                        num_swap_payments,
                        tenor,
                        swap_rate,
                    )
                    .unwrap()
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
        Asset::AMERICANSWAPTION => {
            let SwaptionParameters {
                maturity,
                swap_rate,
                num_swap_payments,
                tenor,
            } = serde_json::from_reader(event.body().as_ref())?;
            let num_tree = 100;
            let func_to_sim = |random_number: f64| {
                let r_t = simulation(random_number);
                hull_white.american_payer_swaption_t(
                    r_t,
                    t,
                    maturity,
                    num_swap_payments,
                    tenor,
                    swap_rate,
                    num_tree,
                )
            };
            Ok(mc(NUM_SIMS, &func_to_sim))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_histogram() {
        let histogram = bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 7.0]);
        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &1);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &2);
    }
    #[test]
    fn test_histogram_edge() {
        let histogram = bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 6.5]);
        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &1);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &2);
    }
    #[test]
    fn test_histogram_edge_2() {
        let histogram = bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 6.499]);
        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &2);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &1);
    }
    #[test]
    fn vasicek_simulation() {
        let r = 0.04;
        let a = 0.3;
        let b = 0.05;
        let sig = 0.001; //to ensure not too great variability
        let t = 50.0;
        let simulation = generate_vasicek(r, a, b, sig, t);
        let n = 500;
        let results = mc_results(n, &simulation);
        let average_result = results.iter().fold(0.0, |a, b| a + b) / (n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result < 0.052, true);
        assert_eq!(average_result > 0.048, true);
    }
    #[tokio::test]
    async fn test_bond_simulation() {
        let input = include_str!("gateway_request.json");
        let request = lambda_http::request::from_str(&input).expect("failed to create request");
        let body = func(request).await.unwrap(); //.body();
        match &body.body() {
            Body::Text(text) => println!("{}", text),
            _ => println!("Not text"),
        }
        assert!(!body.body().is_empty());
    }

    /*#[test]
    fn bond_simulation() {
        let r = 0.04;
        let a = 0.3;
        let b = 0.05;
        let sig = 0.001; //to ensure not too great variability
        let t = transform_days_to_year(10.0);
        let maturity = 1.0;
        let simulation = generate_vasicek(r, a, b, sig, t);
        let yield_fn = yield_curve(r, a, b, sig);
        let forward_fn = forward_curve(r, a, b, sig);
        let func_to_sim = |random_number: f64| {
            let r_t = simulation(random_number);
            hull_white::bond_price_t(r_t, a, sig, t, maturity, &yield_fn, &forward_fn)
        };
        let n = 500;
        let results = mc_results(n, &func_to_sim);
        let average_result = results.iter().fold(0.0, |a, b| a + b) / (n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result < 0.961, true);
        assert_eq!(average_result > 0.960, true);
    }
    #[test]
    fn edf_simulation() {
        let r = 0.04;
        let a = 0.3;
        let b = 0.05;
        let sig = 0.001; //to ensure not too great variability
        let t = transform_days_to_year(10.0);
        let maturity = 1.0;
        let delta = 0.25;
        let simulation = generate_vasicek(r, a, b, sig, t);
        let yield_fn = yield_curve(r, a, b, sig);
        let forward_fn = forward_curve(r, a, b, sig);
        let func_to_sim = |random_number: f64| {
            let r_t = simulation(random_number);
            hull_white.euro_dollar_future_t(r_t, a, sig, t, maturity, delta, &yield_fn, &forward_fn)
        };
        let n = 500;
        let results = mc_results(n, &func_to_sim);
        let average_result = results.iter().fold(0.0, |a, b| a + b) / (n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result < 0.044, true);
        assert_eq!(average_result > 0.042, true);
    }
    #[test]
    fn bondcall_simulation() {
        let r = 0.04;
        let a = 0.3;
        let b = 0.05;
        let sig = 0.001; //to ensure not too great variability
        let t = transform_days_to_year(10.0);
        let maturity = 1.0;
        let bond_maturity = 1.25;
        let strike = 0.97;
        let simulation = generate_vasicek(r, a, b, sig, t);
        let yield_fn = yield_curve(r, a, b, sig);
        let forward_fn = forward_curve(r, a, b, sig);
        let func_to_sim = |random_number: f64| {
            let r_t = simulation(random_number);
            hull_white::bond_call_t(
                r_t,
                a,
                sig,
                t,
                maturity,
                bond_maturity,
                strike,
                &yield_fn,
                &forward_fn,
            )
        };
        let n = 500;
        let results = mc_results(n, &func_to_sim);
        let average_result = results.iter().fold(0.0, |a, b| a + b) / (n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result < 0.019, true);
        assert_eq!(average_result > 0.017, true);
    }
    #[test]
    fn bondput_simulation() {
        let r = 0.04;
        let a = 0.3;
        let b = 0.05;
        let sig = 0.001; //to ensure not too great variability
        let t = transform_days_to_year(10.0);
        let maturity = 1.0;
        let bond_maturity = 1.25;
        let strike = 0.995;
        let simulation = generate_vasicek(r, a, b, sig, t);
        let yield_fn = yield_curve(r, a, b, sig);
        let forward_fn = forward_curve(r, a, b, sig);
        let func_to_sim = |random_number: f64| {
            let r_t = simulation(random_number);
            hull_white::bond_put_t(
                r_t,
                a,
                sig,
                t,
                maturity,
                bond_maturity,
                strike,
                &yield_fn,
                &forward_fn,
            )
        };
        let n = 500;
        let results = mc_results(n, &func_to_sim);
        let average_result = results.iter().fold(0.0, |a, b| a + b) / (n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result < 0.006, true);
        assert_eq!(average_result > 0.005, true);
    }*/
    #[test]
    fn min_v_test() {
        let v = vec![4.0, 2.0, 5.0];
        let result = min_v(&v);
        assert_eq!(result, 2.0);
    }
    #[test]
    fn max_v_test() {
        let v = vec![4.0, 2.0, 5.0];
        let result = max_v(&v);
        assert_eq!(result, 5.0);
    }
}
