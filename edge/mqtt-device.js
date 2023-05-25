const mqtt = require('mqtt');

class MqttDevice {
  constructor(deviceID) {
    this.deviceID = deviceID;
    this.brokerUrl = 'mqtt://test.mosquitto.org';
    this.client = null;
    this.reportingInterval = null;
  }

  connect(callback) {
    this.client = mqtt.connect(this.brokerUrl);

    this.client.on('connect', () => {
      console.log('Connected to MQTT broker');
      callback();
    });

    this.client.on('error', (error) => {
      console.error('MQTT error:', error);
    });

    this.client.on('close', () => {
      console.log('Connection to MQTT broker closed');
    });
  }

  async reportDeviceInfo(deviceInfo, sensorData) {
    const topic = 'sjtu/server/data';
    const message = JSON.stringify({
      deviceID: this.deviceID,
      deviceInfo: deviceInfo,
      sensorData: sensorData,
    });

    this.client.publish(topic, message, (error) => {
      if (error) {
        console.error('Failed to publish message:', error);
      } else {
        console.log('Message published to', topic, message);
      }
    });
  }

  getDeviceInfo(callback) {
    const topic = `sjtu/device/${this.deviceID}/cmd`;

    this.client.subscribe(topic, (error) => {
      if (error) {
        console.error('Failed to subscribe to topic:', error);
      } else {
        console.log('Subscribed to', topic);
      }
    });

    this.client.on('message', (receivedTopic, message) => {
      if (receivedTopic === topic) {
        const command = message.toString();
        if (command === 'device_update') {
          callback(this.deviceID);
        }
      }
    });
  }

  disconnect() {
    if (this.client) {
      this.client.end();
      console.log('Disconnected from MQTT broker');
    }
  }

  start(getDeviceInfoCallback, getSensorDataCallback) {
    this.connect(async () => {
      // 获取设备信息
      await this.polling(getDeviceInfoCallback, getSensorDataCallback);

      // 定时上报设备信息和传感器数据
      this.reportingInterval = setInterval(async () => {
        await this.polling(getDeviceInfoCallback, getSensorDataCallback);
      }, 60000); // 每分钟上报一次
    })
  }

  stop() {
    clearInterval(this.reportingInterval);
    this.disconnect();
  }


  async polling(getDeviceInfoCallback, getSensorDataCallback) {
    // 获取设备信息
    const deviceInfo = await getDeviceInfoCallback();
    const sensorData = await getSensorDataCallback();

    this.reportDeviceInfo(deviceInfo, sensorData);
  }
}

module.exports = MqttDevice;
