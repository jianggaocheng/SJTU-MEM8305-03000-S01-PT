const MqttConnector = require('./mqtt-connector');
const Backend = require('./backend');


const DATABASE = 'backend.db'

// 创建MQTT连接器实例，并连接到MQTT broker
const mqttConnector = new MqttConnector(DATABASE);
mqttConnector.connect();

// 创建HTTP后端实例，并启动HTTP服务器
const httpBackend = new Backend(DATABASE);
httpBackend.start(3000);