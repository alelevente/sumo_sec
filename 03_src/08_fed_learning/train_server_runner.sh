#!/bin/bash

for i in {0..1}
do
    flask --app flask_participant.py run -p $(( $i + 5000)) &
done