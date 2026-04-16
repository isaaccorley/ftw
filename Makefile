.PHONY: install dev build clean check

install:
	bun install

dev:
	bun run dev

build:
	bun run build

check:
	bunx biome check --write src/

clean:
	rm -rf .next out node_modules
