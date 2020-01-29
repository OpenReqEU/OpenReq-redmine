const fs = require("fs");

const Hapi=require('hapi');
const Papa = require("papaparse");
const sw = require("stopword");

process.on('unhandledRejection', r => {throw r;});

const tokenize = (text) => sw.removeStopwords(
    text.toLowerCase().split(/[^a-zA-Z]/).filter(t => t),
    sw.fr);

const jaccard_distance = (tokens1, tokens2) => 
    1 - new Set(tokens1.filter(v => tokens2.includes(v))).size
    / (new Set(tokens1.concat(tokens2))).size;

const tickets = Papa.parse(fs.readFileSync('tickets.csv', 'utf8'), {header: true}).data;
const save_ticket = (ticket) => {
    tickets.push(ticket);
    const csv_data = Papa.unparse(tickets);
    fs.writeFileSync('tickets.csv', csv_data);
};

const server = Hapi.server({ host:'0.0.0.0', port:8000 });

server.route({
    method:'POST',
    path:'/getClosestTicket',
    handler: function (req) {
        const reqText = req.payload.content;
        const reqTokens = tokenize(reqText);
        const reqTicketId = req.payload.ticketId;
        const reqTreshold = req.payload.treshold;
        const reqCount = req.payload.count;
        const reqProject = req.payload.project;

        tickets.sort((a, b) => 
            jaccard_distance(tokenize(a['text']), reqTokens)
            - jaccard_distance(tokenize(b['text']), reqTokens));

        const closests = tickets.slice(0, 6);

        if (!closests.length) {
            save_ticket({ id: reqTicketId, text: reqText, project: reqProject });
            return { closest: {}, project_closests: []}
        }

        const closest = {
            id: closests[0]['id'],
            distance: jaccard_distance(tokenize(closests[0]['text']), reqTokens)
        };

        const project_closests = tickets
            .filter(({project}) => project == reqProject)
            .slice(0, reqCount)
            .filter(({text}) => jaccard_distance(tokenize(text), reqTokens) < reqTreshold)
            .map(({id, text}) => ({ id, distance: jaccard_distance(tokenize(text), reqTokens) }));

        //save the new ticket
        save_ticket({ id: reqTicketId, text: reqText, project: reqProject });
        return { closest, project_closests };
    }
});

(async () => {
    server.route({
        method:'GET',
        path:'/health',
        handler: () => 'OK'
    });
    
    try {
        await server.start();
    }
    catch (err) {
        console.log(err);
        process.exit(1);
    }
    
    console.log('Server running at:', server.info.uri);
    process.on("SIGINT", () => process.exit(0));
})();
