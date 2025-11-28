import requests  # ← YOU WERE MISSING THIS LINE!

# Real password from DECODO
password = "=xah0nQNJfcn85xc3G"  # Starts with equals sign

proxies = {
    'http': f'http://spqtvoaow5:{password}@gate.decodo.com:10001',
    'https': f'http://spqtvoaow5:{password}@gate.decodo.com:10001'
}

try:
    response = requests.get('https://ip.decodo.com/json', proxies=proxies, timeout=10)
    if response.status_code == 200:
        print('✅ SUCCESS! Proxy is working!')
        print(response.text)
    else:
        print(f'❌ HTTP Error: {response.status_code}')
except requests.exceptions.ProxyError:
    print('❌ Proxy connection failed - check credentials')
except Exception as e:
    print(f'❌ Error: {e}')