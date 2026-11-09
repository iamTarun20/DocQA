PYTHON ?= python

.PHONY: test eval demo
test:
	$(PYTHON) -m docqa test

eval:
	$(PYTHON) -m docqa eval

demo:
	$(PYTHON) -m docqa demo
