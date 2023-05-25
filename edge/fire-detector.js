const {PythonShell} = require('python-shell');
const _ = require('lodash');

module.exports = {
  predict: async (imagePath) => {
    let options = {
      mode: 'text',
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: './', // path to your script
      args: [imagePath, './best_model_tl.pth'] // your arguments
    };
    
    let result = await PythonShell.run('predict.py', options);
    let probability = (1 - result).toFixed(3);
    console.log(`Fire Probability: ${probability} `)
    return probability;
  }
}