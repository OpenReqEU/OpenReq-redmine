const fs = require("fs");

const Papa = require("papaparse");
const sw = require("stopword");

process.on('unhandledRejection', r => {throw r;});

const tokenize = (text) => sw.removeStopwords(
    text.toLowerCase().split(/[^a-zA-Z]/).filter(t => t),
    sw.fr);

const jaccard_distance = (tokens1, tokens2) => 
    1 - new Set(tokens1.filter(v => tokens2.includes(v))).size
    / (new Set(tokens1.concat(tokens2))).size;

const data = Papa.parse(fs.readFileSync('data/ref_subset.csv', 'utf8'), {header: true}).data;
const tokenized = data.map(({id, text}) => ([id, tokenize(text)]));

const result = [...tokenized].map(([id, text], i) => {
    console.log(100 * i / tokenized.length);
    tokenized.sort((a, b) => jaccard_distance(a[1], text) - jaccard_distance(b[1], text));

    const closest = tokenized.slice(0, 6).filter(t => t[0] != id);
    
    return {
        id,
        closest_ids: closest.map(c => c[0]),
        distance: jaccard_distance(closest[0][1], text)
    }
});

fs.writeFileSync("result_subset.json", JSON.stringify(result));