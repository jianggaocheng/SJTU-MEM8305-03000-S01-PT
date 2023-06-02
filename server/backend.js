const express = require('express');
const sqlite3 = require('sqlite3');
const path = require('path');
const MqttConnector = require('./mqtt-connector');

class Backend {
  constructor(databasePath) {
    this.db = new sqlite3.Database(databasePath);
    this.mqttConnector = new MqttConnector(databasePath);
  }

  start(port) {
    const app = express();

    app.use(express.json());

    app.get('/devices', (req, res) => {
      // 查询设备列表
      this.db.all('SELECT * FROM devices', (error, rows) => {
        if (error) {
          console.error('Failed to fetch devices:', error);
          res.status(500).json({ error: 'Failed to fetch devices' });
        } else {
          res.json(rows);
        }
      });
    });

    app.post('/devices/:deviceID/command', (req, res) => {
      const deviceID = req.params.deviceID;
      const command = req.body.command;

      // 在这里处理接收到的指令，例如将指令发送给设备
      console.log(`Received command ${command} for device ${deviceID}`);

      // 调用 MQTT 模块的 pushCommand 方法发送指令
      this.mqttConnector.pushCommand(deviceID, command);

      res.json({ message: 'Command sent successfully' });
    });

    // 托管前端页面
    const publicPath = path.join(__dirname, 'public');
    app.use(express.static(publicPath));

    app.listen(port, () => {
      console.log(`HTTP server started on port ${port}`);
    });
  }
}

module.exports = Backend;