import os
import requests
import random
import hashlib
import string


def translate_text_baidu(text: str) -> str:
    api_url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    appid = os.getenv("BAIDUFANYI_APPID")
    secret_key = os.getenv("BAIDUFANYI_SECRET_KEY")
    from_lang = "zh"
    to_lang = "en"
    salt = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    sign = appid + text + salt + secret_key
    md5 = hashlib.md5()
    md5.update(sign.encode('utf-8'))
    sign = md5.hexdigest()

    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": appid,
        "salt": salt,
        "sign": sign
    }

    response = requests.get(api_url, params=params)
    response_json = response.json()

    if "trans_result" in response_json:
        return response_json["trans_result"][0]["dst"]
    else:
        return "Error: " + response_json["error_msg"]