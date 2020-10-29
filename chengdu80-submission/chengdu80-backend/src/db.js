import { initModels } from './models/init-models';

const config = {
  HOST: '18.141.37.167',
  USER: 'postgres',
  PASSWORD: 'postgres',
  DB: 'cd80',
  dialect: 'postgres',
  pool: {
    max: 5,
    min: 0,
    acquire: 30000,
    idle: 10000,
  },
};

const Sequelize = require('sequelize');
const sequelize = new Sequelize(
  config.DB,
  config.USER,
  config.PASSWORD,
  {
    host: config.HOST,
    dialect: config.dialect,
    operatorsAliases: false,

    pool: {
      max: config.pool.max,
      min: config.pool.min,
      acquire: config.pool.acquire,
      idle: config.pool.idle,
    },
  },
);

const db = {};

db.Sequelize = Sequelize;
db.sequelize = sequelize;

// init all models
db.models = initModels(sequelize);

module.exports = db;
