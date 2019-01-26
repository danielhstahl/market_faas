#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate serde_json;
extern crate rand;
use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
extern crate lambda_http;
extern crate lambda_runtime as runtime;
extern crate hull_white;
extern crate rayon;
use std::f64;
use self::rayon::prelude::*;
use lambda_http::{lambda, Body, IntoResponse, Request, Response, RequestExt};
use runtime::{error::HandlerError, Context};
use std::error::Error;
use std::collections::HashMap;

fn build_response(code: u16, body: &str) -> impl IntoResponse {
    Response::builder()
        .status(code)
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Credentials", "true")
        .body::<Body>(body.into())
        .unwrap()
}
fn construct_error(e_message: &str) -> String {
    json!({ "err": e_message }).to_string()
}

fn yield_curve(curr_rate:f64, a:f64, b:f64, sig:f64)->impl Fn(f64)->f64{
    move |t|{
        let at=(1.0-(-a*t).exp())/a;
        let ct=(b-sig.powi(2)/(2.0*a.powi(2)))*(at-t)-sig.powi(2)*at.powi(2)/(4.0*a);
        at*curr_rate-ct
    }
}
fn forward_curve(curr_rate:f64, a:f64, b:f64, sig:f64)->impl Fn(f64)->f64{
    move |t|{
        let tmp=(-a*t).exp();
        b+tmp*(curr_rate-b)-(sig.powi(2)/(2.0*a.powi(2)))*(1.0-tmp).powi(2)
    }
}

fn generate_vasicek(curr_rate:f64, a:f64, b:f64, sig:f64, t:f64)->impl Fn(f64)->f64{
    let tmp=(-a*t).exp();
    let mu=b*(1.0-tmp)+curr_rate*tmp;
    let vol=sig*((1.0-(-2.0*a*t).exp())/(2.0*a)).sqrt();
    move |random_number|{
        mu+vol*random_number
    }
}

const NUM_SIMS:usize=500;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BondParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EDFParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64,
    tenor: f64
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BondOptionParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64,
    underlying_maturity: f64,
    strike: f64
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CapletParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64,
    tenor: f64,
    strike: f64
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwapParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64,
    tenor: f64,
    swap_rate: f64
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SwaptionParameters {
    t: f64, //in days
    r0: f64,
    a:f64,
    b:f64,
    sigma:f64,
    maturity:f64,
    tenor: f64,
    swap_tenor:f64,
    swap_rate: f64
}

fn bin(min:f64, max:f64, num_bins:f64, elements:&[f64])->HashMap<String, usize>{
    let mut bins=HashMap::new();
    let range=max-min;
    let bin_width=range/num_bins;
    for element in elements.iter(){
        let key=if element==&max{
            format!("{:.4}-{:.4}", max-bin_width, max)
        }
        else {
            let lower_index=((element-min)/bin_width).floor();
            let lower_bound=lower_index*bin_width+min;
            let upper_bound=(lower_index+1.0)*bin_width+min;
            format!("{:.4}-{:.4}", lower_bound, upper_bound)
        };
        if let Some(x) = bins.get_mut(&key) {
            *x +=1;
        }
        else{
            bins.insert(key, 1);
        }
    }
    bins
}

fn combine_and_bin(min:f64, max:f64, elements:&[f64])->HashMap<String, usize>{
    let num_bins=(2.0*(elements.len() as f64).powf(1.0/3.0)).floor();
    bin(min, max, num_bins, elements)
}

fn main() -> Result<(), Box<dyn Error>> {
    lambda!(market_faas_wrapper);
    Ok(())
}
fn market_faas_wrapper(event: Request, _ctx: Context) -> Result<impl IntoResponse, HandlerError> {
    match market_faas(event) {
        Ok(res) => Ok(build_response(200, &json!(res).to_string())),
        Err(e) => Ok(build_response(400, &construct_error(&e.to_string()))),
    }
}

fn mc_results<T>(num_sims:usize, func_to_sim:T)->Vec<f64>
where T:Fn(f64)->f64+std::marker::Sync
{
    let normal = StandardNormal;
    (0..num_sims).into_par_iter()
        .map(|_index|{
            let norm = thread_rng().sample(normal);
            func_to_sim(norm)
        }).collect()
}

fn mc<T>(num_sims:usize, func_to_sim:T)->HashMap<String, usize>
where T:Fn(f64)->f64+std::marker::Sync
{
    let results=mc_results(num_sims, &func_to_sim);
    let min = results.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = results.iter().fold(f64::NEG_INFINITY, |a, &b| a.min(b));
    combine_and_bin(min, max, &results)
}

//simplistic, but good enough
fn transform_days_to_year(t:f64)->f64{
    t/365.0
}
fn bond(event:Request)->Result<HashMap<String, usize>, Box<dyn Error>>{
    let BondParameters {
        t, r0, a, b, maturity, sigma
    } = serde_json::from_reader(event.body().as_ref())?;
    let t=transform_days_to_year(t);
    let yield_fn=yield_curve(r0, a, b, sigma);
    let forward_fn=forward_curve(r0, a, b, sigma);
    let simulation=generate_vasicek(r0, a, b, sigma, t);
    let func_to_sim=|random_number:f64|{
        let r_t=simulation(random_number);
        hull_white::bond_price_t(
            r_t, a, sigma, t, 
            maturity, &yield_fn, 
            &forward_fn
        )
    };
    Ok(mc(NUM_SIMS, &func_to_sim))
}
fn market_faas(event: Request) -> Result<HashMap<String, usize>, Box<dyn Error>> {
    let path_parameters = event.path_parameters();
    let asset=path_parameters.get("asset").unwrap_or("bond");
    let histogram=match asset{
        
        "bond"=>{
            bond(event)?
        },
        "edf"=>{
            let EDFParameters {
                t, r0, a, b, maturity, sigma, tenor
            } = serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::euro_dollar_future_t(
                    r_t, a, sigma, t, 
                    maturity, tenor, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        },
        "bondcall"=>{
            let BondOptionParameters{
                t, r0, a, b, maturity, 
                sigma, underlying_maturity, strike
            } = serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::bond_call_t(
                    r_t, a, sigma, t, 
                    maturity, underlying_maturity, 
                    strike, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        },
        "bondput"=>{
            let BondOptionParameters{
                t, r0, a, b, maturity, 
                sigma, underlying_maturity, strike
            } = serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::bond_put_t(
                    r_t, a, sigma, t, 
                    maturity, underlying_maturity, 
                    strike, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        },
        "caplet"=>{
            let CapletParameters{
                t, r0, a, b, maturity, 
                sigma, strike, tenor
            } = serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::caplet_t(
                    r_t, a, sigma, t, 
                    maturity, tenor, 
                    strike, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        },
        "swap"=>{
            let SwapParameters{
                t, r0, a, b, maturity, 
                sigma, swap_rate, tenor
            }= serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::swap_price_t(
                    r_t, a, sigma, t, 
                    maturity, tenor, 
                    swap_rate, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        }, 
        "swaption"=>{
            let SwaptionParameters{
                t, r0, a, b, maturity, 
                sigma, swap_rate, tenor, swap_tenor
            }= serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::european_payer_swaption_t(
                    r_t, a, sigma, t, 
                    swap_tenor, 
                    maturity, tenor, 
                    swap_rate, &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        }, 
        "americanswaption"=>{
            let SwaptionParameters{
                t, r0, a, b, maturity, 
                sigma, swap_rate, tenor, swap_tenor
            }= serde_json::from_reader(event.body().as_ref())?;
            let t=transform_days_to_year(t);
            let yield_fn=yield_curve(r0, a, b, sigma);
            let forward_fn=forward_curve(r0, a, b, sigma);
            let simulation=generate_vasicek(r0, a, b, sigma, t);
            let num_tree=100;
            let func_to_sim=|random_number:f64|{
                let r_t=simulation(random_number);
                hull_white::american_payer_swaption_t(
                    r_t, a, sigma, t, 
                    swap_tenor, 
                    maturity, tenor, 
                    swap_rate, 
                    num_tree,
                    &yield_fn, 
                    &forward_fn
                )
            };
            mc(NUM_SIMS, &func_to_sim)
        },
        _=>{ //I wish the compiler was smarter than this...we know it wont get here
            bond(event)?
        }
    };
    Ok(histogram)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_histogram() {
        let histogram=bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 7.0]);
       /* for (key, value) in &histogram{
            println!("this is key: {}", key);
            println!("this is value: {}",value);
        }*/

        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &1);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &2);
    }
    #[test]
    fn test_histogram_edge() {
        let histogram=bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 6.5]);
        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &1);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &2);
    }
    #[test]
    fn test_histogram_edge_2() {
        let histogram=bin(5.0, 8.0, 2.0, &vec![5.0, 8.0, 6.499]);
        assert_eq!(histogram.contains_key("5.0000-6.5000"), true);
        assert_eq!(histogram.contains_key("6.5000-8.0000"), true);
        assert_eq!(histogram.get("5.0000-6.5000").unwrap(), &2);
        assert_eq!(histogram.get("6.5000-8.0000").unwrap(), &1);
    }
    #[test]
    fn vasicek_simulation(){
        let r=0.04;
        let a=0.3;
        let b=0.05;
        let sig=0.001; //to ensure not too great variability
        let t=50.0;
        let simulation=generate_vasicek(r, a, b, sig, t);
        /*let func_to_sim=|random_number:f64|{
            let r_t=simulation(random_number);
            hull_white::bond_price_t(
                r_t, a, sigma, t, 
                maturity, &yield_fn, 
                &forward_fn
            )
        };*/
        let n=500;
        let results=mc_results(n, &simulation);
        let average_result=results.iter().fold(0.0, |a, b|a+b)/(n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result<0.052, true);
        assert_eq!(average_result>0.048, true);

    }
    #[test]
    fn bond_simulation(){
        let r=0.04;
        let a=0.3;
        let b=0.05;
        let sig=0.001; //to ensure not too great variability
        let t=10.0/365.0;
        let maturity=50.0;
        let simulation=generate_vasicek(r, a, b, sig, t);
        let yield_fn=yield_curve(r, a, b, sig);
        let forward_fn=forward_curve(r, a, b, sig);
        let func_to_sim=|random_number:f64|{
            let r_t=simulation(random_number);
            hull_white::bond_price_t(
                r_t, a, sig, t, 
                maturity, &yield_fn, 
                &forward_fn
            )
        };
        let n=500;
        let results=mc_results(n, &func_to_sim);
        let average_result=results.iter().fold(0.0, |a, b|a+b)/(n as f64);
        println!("this is average: {}", average_result);
        assert_eq!(average_result<0.052, true);
        assert_eq!(average_result>0.048, true);

    }
}
