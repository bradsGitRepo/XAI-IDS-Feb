#!/usr/bin/env python3

"""
Configuration file for the IDS system.
Contains whitelists for both development and production environments.
"""

# Development environment whitelist - used for testing and development
DEV_WHITELIST = [
    ('192.168.0.102', 'f4:b8:98:8b:da:b9'),
    ('192.168.0.103', 'f4:b8:98:a4:91:31'),
    ('192.168.0.104', '34:08:e1:ba:f1:f7'),
    ('192.168.0.105', 'f4:b8:98:a1:28:be'),
    ('192.168.0.106', '34:08:e1:d0:09:87'),
    ('192.168.0.107', 'f4:b8:98:90:51:a2'),
    ('192.168.0.130', '94:f3:92:56:48:77'),
    ('192.168.5.131', '94:f3:92:56:48:77'),
    ('192.168.0.128', 'ac:71:2e:93:3f:be'),
    ('192.168.0.110', '30:10:E4:AC:EE:2D'),
]

# Production environment whitelist - actual train-line cyber-physical system
PROD_WHITELIST = [
    ('192.168.0.102', 'f4:b8:98:8b:da:b9'),
    ('192.168.0.103', 'f4:b8:98:a4:91:31'),
    ('192.168.0.104', '34:08:e1:ba:f1:f7'),
    ('192.168.0.105', 'f4:b8:98:a1:28:be'),
    ('192.168.0.106', '34:08:e1:d0:09:87'),
    ('192.168.0.107', 'f4:b8:98:90:51:a2'),
    ('192.168.0.130', '94:f3:92:56:48:77'),
    ('192.168.5.131', '94:f3:92:56:48:77'),
    ('192.168.0.128', 'ac:71:2e:93:3f:be'),
    ('192.168.0.110', '30:10:E4:AC:EE:2D'),
]

def get_whitelist(environment=None):
    """
    Get the appropriate whitelist based on the environment.
    
    Args:
        environment (str): Either 'dev' or 'prod'. If None, will check IDS_ENV environment variable.
                         Defaults to 'dev' if not specified.
    
    Returns:
        set: Set of (ip, mac) tuples representing whitelisted devices
    """
    import os
    
    if environment is None:
        environment = os.getenv('IDS_ENV', 'dev')
    
    if environment.lower() == 'prod':
        return set(PROD_WHITELIST)
    return set(DEV_WHITELIST)
