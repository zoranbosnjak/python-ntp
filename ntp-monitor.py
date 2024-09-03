#!/usr/bin/env python3

import argparse
from time import sleep

import ntp

max_poll = 300.0

def main():
    parser = argparse.ArgumentParser(description='NTP server monitor.')
    parser.add_argument('ntp', nargs='+', help='NTP server IP address')
    args = parser.parse_args()

    ntp_arena = ntp.NtpArena(addresses=args.ntp, max_poll=max_poll)

    while True:
        print('---')
        try:
            pause = ntp_arena.query_peers()
            _leap, offset, _jitter = ntp_arena.calculate_state()
            print('offset: {}'.format(offset))
            for (peer_name, peer_obj) in ntp_arena.peers.items():
                name = peer_name[0]
                fit = peer_obj.is_fit()
                peer_offset = peer_obj.state.offset if fit else None
                print('peer: {}, offset: {}'.format(name, peer_offset))
        except ntp.NtpError as e:
            print('problem...', e)
            pause = max_poll
        sleep(pause)

if __name__ == "__main__":
    main()
