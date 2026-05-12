#!/usr/bin/env sh

cat assets/**/*-train.conllu > assets/train.conllu
cat assets/**/*-dev.conllu > assets/dev.conllu
cat assets/**/*-test.conllu > assets/test.conllu