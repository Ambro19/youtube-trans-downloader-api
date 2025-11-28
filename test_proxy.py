# Create a test script

import requests

proxies = {
    'http': 'http://spqtvoaow5:→xah0nQNJfcn85xc3Q@gate.decodo.com:10001',
    'https': 'http://spqtvoaow5:→xah0nQNJfcn85xc3Q@gate.decodo.com:10001'
}

try:
    response = requests.get('https://ip.decodo.com/json', proxies=proxies, timeout=10)
    print('✅ Proxy working!')
    print(response.text)
except Exception as e:
    print(f'❌ Proxy error: {e}')
"@ | Out-File -FilePath test_proxy.py -Encoding UTF8"

