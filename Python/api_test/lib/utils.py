import time
import random
import string

def get_timestamp():
    return str(int(time.time()))

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))