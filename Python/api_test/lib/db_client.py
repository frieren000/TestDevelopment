# 数据库客户端
import os
import yaml
import pymysql
import logging

logger = logging.getLogger(__name__)


class DBClient:
    def __init__(self, config_path='config/config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"数据库配置文件不存在: {os.path.abspath(config_path)}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        db_conf = config.get('db')
        if not db_conf:
            raise KeyError("配置文件中缺少 'db' 节点")

        self.connection_params = {
            'host': db_conf.get('host', 'localhost'),
            'port': int(db_conf.get('port', 3306)),
            'user': db_conf['user'],
            'password': db_conf['password'],
            'database': db_conf['database'],
            'charset': 'utf8mb4',
            'autocommit': False,
            'cursorclass': pymysql.cursors.DictCursor  # 返回字典而非元组
        }

        logger.info(f"DBClient 初始化完成 | Host: {self.connection_params['host']} | DB: {self.connection_params['database']}")

    def _get_connection(self):
        """获取新连接"""
        return pymysql.connect(**self.connection_params)

    def query_one(self, sql, params=None):
        """
        执行 SELECT 查询，返回第一行（字典格式）
        :param sql: SQL 语句，使用 %s 作为占位符
        :param params: 参数元组或字典
        :return: dict 或 None
        """
        with self._get_connection() as cursor:
            logger.debug(f"→ QUERY ONE: {sql} | Params: {params}")
            cursor.execute(sql, params)
            result = cursor.fetchone()
            logger.debug(f"← RESULT: {result}")
            return result

    def query_all(self, sql, params=None):
        """
        执行 SELECT 查询，返回所有行
        :return: list[dict]
        """
        with self._get_connection() as cursor:
            logger.debug(f"→ QUERY ALL: {sql} | Params: {params}")
            cursor.execute(sql, params)
            result = cursor.fetchall()
            logger.debug(f"← ROWS: {len(result)}")
            return result

    def execute(self, sql, params=None):
        """
        执行 INSERT / UPDATE / DELETE 语句
        :return: 受影响的行数
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.debug(f"→ EXECUTE: {sql} | Params: {params}")
                affected = cursor.execute(sql, params)
                conn.commit()
                logger.debug(f"← COMMITTED | Affected rows: {affected}")
                return affected
        except Exception as e:
            conn.rollback()
            logger.error(f"SQL 执行失败，已回滚: {e}")
            raise
        finally:
            conn.close()

    def execute_many(self, sql, params_list):
        """
        批量执行（如批量插入）
        :param sql: SQL 模板
        :param params_list: 参数列表
        :return: 总影响行数
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                logger.debug(f"→ EXECUTE MANY: {sql} | Count: {len(params_list)}")
                affected = cursor.executemany(sql, params_list)
                conn.commit()
                logger.debug(f"← BATCH COMMITTED | Affected rows: {affected}")
                return affected
        except Exception as e:
            conn.rollback()
            logger.error(f"批量 SQL 执行失败，已回滚: {e}")
            raise
        finally:
            conn.close()