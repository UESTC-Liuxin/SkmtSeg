import os
import datetime
import logging

def get_logger(logdir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ts = str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    ts = ts.replace(':', '_').replace('-', '_')
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))

    hdlr = logging.FileHandler(file_path)
    hdlr.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()  # 日志控制台输出
    stream_handler.setLevel(logging.NOTSET)


    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(hdlr)

    return logger


if __name__ =="__main__":
    logger=get_logger(".././")
    logger.info("aaa")