from flask import request, app


@app.route('/user')
def find_users_noncompliant():
    from flask import request
    import xml.etree.ElementTree as ET
    tree = ET.parse('users.xml')
    root = tree.getroot()
    username = request.args['username']
    query = "./users/user/[@name='"+username+"']/location"
    # Noncompliant: evaluating expression built from user-supplied parameter
    # can lead to XPath injection.
    elements = root.findall(query)
    return 'Location %s' % list(elements)



def create_connection_noncompliant():
    import socket
    import ssl
    host, port = 'example.com', 443
    with socket.socket(socket.AF_INET) as sock:
        context = ssl.SSLContext()
        # Noncompliant: security certificate validation disabled.
        context.verify_mode = ssl.CERT_NONE
        conn = context.wrap_socket(sock, server_hostname=host)
        try:
            conn.connect((host, port))
            handle(conn)
        finally:
            conn.close()


@app.route('/someUrl')
def zip_bomb_attack_noncompliant():
    file = request.files['file']
    filename = file.filename
    file.save(filename)
    tfile = tarfile.open(filename)
    # Noncompliant: Untrusted archive file extracted without any validation.
    tfile.extractall('./tmp/')
    tfile.close()


def log_credentials_noncompliant():
    import boto3
    import logging
    session = boto3.Session()
    credentials = session.get_credentials()
    credentials = credentials.get_frozen_credentials()
    access_key = credentials.access_key
    secret_key = credentials.secret_key
    # Noncompliant: credentials are written to the logger.
    logging.info('Access key: ', access_key)
    logging.info('secret access key: ', secret_key)
