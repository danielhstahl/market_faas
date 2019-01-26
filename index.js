const yamlWrite=require('./src/writeYml')

yamlWrite().catch(err=>{
    console.log(err)
})