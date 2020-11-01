# Chengdu80: Pisces Team

## Details
- To access: http://18.141.37.167:8081/

- Ports:
    - 5432: Postgress
    - 8080: Backend (~/chengdu80-backend/)
    - 8081: Frontend (~/chengdu80-frontend/)

### Postgres
```
$ systemctl start postgresql
```

### Frontend
```
cd /home/ubuntu/chengdu80-frontend
$ npm run start &
```

### Backend
```
cd /home/ubuntu/chengdu80-backend
$ npm run start &
```
