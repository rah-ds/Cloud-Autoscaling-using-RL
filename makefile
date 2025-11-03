
# useful for debugging
# default:
# 	echo @ makefile

include .env
export

## runs ruff to auto format code
auto_format:
	ruff format src/*.py scripts/*.py

## builds the python package `cloud_autoscaling_rl` in editable mode
build_src:
	uv pip install -e .

## check readme
check_readme:
	proselint README.md;
	cspell README.md;

## journal lint and save
check_save_journal:
	proselint journals/journal.md;
# 	cspell journals/journal.md;
	pandoc journals/journal.md -o docs/current_journal.docx;
	open docs/current_journal.docx;

## runs ruff to check code style
lint:
	ruff check src/*.py scripts/*.py

## runs the package tests
tests: # only non integration tests
	source .venv/bin/activate && pytest tests -vvx -p no:warnings

## cz release - still needs push
release:
	cz commit;
	cz bump -yes;

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
