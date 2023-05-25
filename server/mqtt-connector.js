const mqtt = require('mqtt');
const sqlite3 = require('sqlite3');

class MqttConnector {
  constructor(databasePath) {
    this.brokerUrl = 'mqtt://test.mosquitto.org';
    this.client = null;
    this.db = new sqlite3.Database(databasePath);

    // 初始化数据库
    this.initializeDatabase();
  }

  initializeDatabase() {
    this.db.serialize(() => {
      // 创建设备表
      this.db.run(`
        CREATE TABLE IF NOT EXISTS devices (
          deviceID TEXT PRIMARY KEY,
          lastConnectionTime TIMESTAMP,
          sensorData INTEGER
        )
      `);

      // 创建传感器数据表
      this.db.run(`
        CREATE TABLE IF NOT EXISTS sensorData (
          deviceID TEXT,
          timestamp TIMESTAMP,
          sensorValue INTEGER,
          FOREIGN KEY (deviceID) REFERENCES devices(deviceID)
        )
      `);
    });
  }

  connect() {
    this.client = mqtt.connect(this.brokerUrl);

    this.client.on('connect', () => {
      console.log('Connected to MQTT broker');

      // 订阅设备传来的消息
      this.client.subscribe('sjtu/server/data', (error) => {
        if (error) {
          console.error('Failed to subscribe to topic:', error);
        } else {
          console.log('Subscribed to topic: sjtu/server/data');
        }
      });
    });

    this.client.on('message', (topic, message) => {
      // 解析消息
      const data = JSON.parse(message.toString());
      const deviceID = data.deviceID;
      const timestamp = new Date().toISOString();
      const sensorData = JSON.stringify(data.sensorData);

      console.log(`Received message: ${JSON.stringify(data)}`);

      // 更新或插入设备记录
      this.db.run(`
        INSERT OR REPLACE INTO devices (deviceID, lastConnectionTime, sensorData)
        VALUES (?, ?, ?)
      `, [deviceID, timestamp, sensorData], (error) => {
        if (error) {
          console.error('Failed to update device record:', error);
        } else {
          console.log('Device record updated:', deviceID);
        }
      });

      // 保存传感器数据
      this.db.run(`
        INSERT INTO sensorData (deviceID, timestamp, sensorValue)
        VALUES (?, ?, ?)
      `, [deviceID, timestamp, sensorData], (error) => {
        if (error) {
          console.error('Failed to insert sensor data:', error);
        } else {
          console.log('Sensor data inserted:', deviceID, timestamp, sensorData);
        }
      });
    });

    this.client.on('error', (error) => {
      console.error('MQTT error:', error);
    });

    this.client.on('close', () => {
      console.log('Connection to MQTT broker closed');
    });
  }

  pushCommand(deviceID, command) {
    const topic = `sjtu/device/${deviceID}/cmd`;
    const message = command;

    this.client.publish(topic, message, (error) => {
      if (error) {
        console.error('Failed to push command:', error);
      } else {
        console.log('Command pushed to device:', deviceID);
      }
    });
  }
}

module.exports = MqttConnector;