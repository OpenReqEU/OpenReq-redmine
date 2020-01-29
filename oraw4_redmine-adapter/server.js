const Hapi = require('@hapi/hapi');
const fetch = require('node-fetch');

process.on('unhandledRejection', r => {throw r;});

const server = Hapi.server({ host:'0.0.0.0', port:8000 });

server.route({
    method:'GET',
    path:'/getDomain',
    handler: async function (req) {
        let result;

        const params = new URLSearchParams(Object.entries({ 
            model_id: '0', 
            text: req.query.t + ' ' + req.query.b
        }));

        const aw_class_id = await fetch('http://oraw4_aw:8080/predict?' + params)
            .then(r => r.json());
        
        if (aw_class_id == 0) {
            result = 'installation-acces-mail';
        } else if (aw_class_id == 1) {
            result = 'ingogérence-service';
        } else if (aw_class_id == 2) {
            result = 'supervision-anomalies';
        } else if (aw_class_id == 3) {
            result = 'web';
        } else {
            result = 'mise-à-jour-sécurité';
        }

        return {'topScoringIntent': {'intent': result}};
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
