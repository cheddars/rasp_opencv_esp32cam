from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from configparser import ConfigParser

config = ConfigParser()
cfg = config.read('config.ini')

if not cfg:
    raise Exception('Config file(config.ini) not found')

host = config.get('db', 'host')
user = config.get('db', 'user')
password = config.get('db', 'password')
dbname = config.get('db', 'dbname')

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{dbname}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()