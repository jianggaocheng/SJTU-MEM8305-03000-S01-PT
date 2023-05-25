const {PythonShell} = require('python-shell');

let options = {
    mode: 'text',
    pythonOptions: ['-u'], // get print results in real-time
    scriptPath: './', // path to your script
    args: ['./fire.png', './best_model_tl.pth'] // your arguments
};

PythonShell.run('predict.py', options, function (err, results) {
    if (err) {
        console.error('Python error: ', err);
        throw err;
    }
    if (results && results.length > 0) {
        console.log('Fire Probability: ', results[0]);
    } else {
        console.log('No result returned from python script');
    }
});
