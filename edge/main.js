const fireDetector = require('./fire-detector');
const MqttDevice = require('./mqtt-device');

(async ()=> {    
    // 创建设备实例，传入设备ID
    const deviceID = 'test1';
    const device = new MqttDevice(deviceID);

    // 回调函数来获取设备信息和传感器数据
    const getDeviceInfo = async () => {
        return {
            name: "device1",
            "serialNumber": "123"
        }
    };

    const getSensorData = async () => {
        let value = await fireDetector.predict('./fire2.jpg');
        return {
            "probability": value
        }
    };

    // 启动设备连接和定时上报
    device.start(getDeviceInfo, getSensorData);
})();
