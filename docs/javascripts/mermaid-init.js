// Material for MkDocs: initialize Mermaid diagrams on every page change.
// Docs: https://squidfunk.github.io/mkdocs-material/reference/diagrams/
const mermaid = window.mermaid;
if (mermaid) {
  mermaid.initialize({ startOnLoad: false });
  // `document$` is provided by the Material theme for client-side navigation.
  // eslint-disable-next-line no-undef
  document$.subscribe(() => {
    mermaid.init(undefined, document.querySelectorAll(".mermaid"));
  });
}
