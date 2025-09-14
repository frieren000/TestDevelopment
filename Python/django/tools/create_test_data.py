import csv
import pathlib
import pymysql
from faker import Faker
from base_tools import *

def generate_data(file_name, data_amount, writer_title_list, faker_dict):
    # 生成一定的测试数据
    csv_file_path = ''
    with open(f'{file_name}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(writer_title_list)
        for _ in range(data_amount):
            writer.writerow(faker_dict[col]() for col in writer_title_list)
            
    root_path = pathlib.Path('.')
    for item in root_path.iterdir():
        if item.is_file() and item.suffix == '.csv' and item.stem == file_name:
            csv_file_path = item.resolve()
            
    print(f"{data_amount}条用户数据已写入{file_name}.csv, 文件位置:{csv_file_path}")
    return csv_file_path

def write_database(csv_file_path, file_name, title_list):
    connect = pymysql.connect(**database_config)
    cursor = connect.cursor()

    try:
        column_defs = []
        for col_name in title_list:
            safe_col_name = f"`{col_name}`"
            col_type = "INTEGER" if "age" in col_name or "id" in col_name.lower() else "TEXT"
            column_defs.append(f"{safe_col_name} {col_type}")

        columns_sql = ",\n        ".join(column_defs)
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{file_name}` (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            {columns_sql}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_sql)

        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            placeholders = ', '.join(['%s'] * len(title_list))
            safe_columns = [f"`{col}`" for col in title_list]
            columns_str = ', '.join(safe_columns)
            insert_sql = f"INSERT INTO `{file_name}` ({columns_str}) VALUES ({placeholders})"

            for row in reader:
                cursor.execute(insert_sql, row)

        connect.commit()
        print(f"表`{file_name}` 创建并写入数据成功！")

    except Exception as e:
        print(f"操作失败：{e}")
        connect.rollback()
        return -1
    finally:
        cursor.close()
        connect.close()

if __name__ == '__main__':
    file_name = 'users'
    data_amount = 100
    writer_title_list = []
    title_list = ['name', 'phone_num', 'email', 'company', 'job', 'ipv4']

    fake = Faker('zh_CN')
    faker_dict = {'name': lambda: fake.name(), 'phone_num': lambda: fake.phone_number(),
                'email': lambda: fake.email(), 'company': lambda: fake.company(),
                'job': lambda: fake.job(), 'ipv4': lambda: fake.ipv4(),
                }



    for i in range(len(title_list)):
        writer_title_list.append(title_list[i])
        
    csv_file_path = generate_data(file_name, data_amount, writer_title_list, faker_dict)
    write_database(csv_file_path, file_name, title_list)

