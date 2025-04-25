# 🦌 DeerFlow Web UI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Come from Open Source, Back to Open Source

This is the web UI for [`DeerFlow`](https://github.com/bytedance/deer-flow).

[`DeerFlow`](https://github.com/bytedance/deer-flow) is a community-driven AI automation framework that builds upon the incredible work of the open source community. Our goal is to combine language models with specialized tools for tasks like web search, crawling, and Python code execution, while giving back to the community that made this possible.

## How to Install

DeerFlow Web UI uses `pnpm` as package manager.

```bash
cd web
pnpm install
```

## How to Run in Development Mode

**Note**: You need to start the Python API service before running the web UI.

Start the web UI development server:

```bash
cd web
pnpm dev
```

By default, the web UI will be available at `http://localhost:3000`.

You can set the `NEXT_PUBLIC_API_URL` environment variable if you're using a different host or location.

```ini
# .env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```


## License

This project is open source and available under the [MIT License](../LICENSE).

## Acknowledgments

Special thanks to all the open source projects and contributors that make `DeerFlow` possible. We stand on the shoulders of giants.

In particular, we want to express our deep appreciation for:
* [Next.js](https://nextjs.org/) for their exceptional framework
* [Shadcn](https://ui.shadcn.com/) for their minimalistic components that powers our UI
* [Zustand](https://zustand.docs.pmnd.rs/) for their stunning state management
* [Framer Motion](https://www.framer.com/motion/) for their amazing animation library
* [React Markdown](https://www.npmjs.com/package/react-markdown) for their exceptional markdown rendering and customizability
* Last but not least, special thanks to [SToneX](https://github.com/stonexer) for his great contribution for [token-by-token visual effect](./src/core/rehype/rehype-split-words-into-spans.ts)

These outstanding projects form the backbone of DeerFlow and exemplify the transformative power of open source collaboration.
