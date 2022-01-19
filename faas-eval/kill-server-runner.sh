#!/bin/sh

pid=$(ps aux | tr -s ' ' | grep server-runner | head -n1 | cut -f2 -d' ')
kill $pid

