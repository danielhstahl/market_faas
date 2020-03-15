const refparser = require("@apidevtools/json-schema-ref-parser")
const YAML = require('yaml')
const fs = require('fs')

module.exports = () => Promise.resolve(() => {
    const root = YAML.parse(fs.readFileSync('./docs/openapi.yml').toString())
    refparser.dereference(root).then((parsed) => {
        fs.writeFileSync(
            './docs/openapi_merged.yml',
            YAML.stringify(parsed)
        )
    })
})