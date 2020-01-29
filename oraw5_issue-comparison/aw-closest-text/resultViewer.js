const fs = require("fs").promises;

const readlineAsyncGenerator = require("readline-async-generator");

(async function(){
    const lineIt = readlineAsyncGenerator();
    const data = Papa.parse(await fs.readFile('data/15ke_clean.csv', 'utf8')).data;
    const results = require("./result.json");

    for (const row of results) {
        await lineIt.next();
        process.stdout.write('\033c');
        console.log(">>> " + row.id);
        console.log(data.find(d => d[0] === row.id)[1]);
        console.log(">>> " + row.closest, row.distance);
        console.log(data.find(d => d[0] === row.closest)[1]);
    }
})();