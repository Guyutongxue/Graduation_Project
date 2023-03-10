import * as monaco from "monaco-editor";
import transpilerDecl from "transpiler/client-type?raw";
import utilsDecl from "rtlib/client-type?raw";

// https://github.com/microsoft/monaco-editor/blob/main/docs/integrate-esm.md#using-vite
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

// @ts-ignore
self.MonacoEnvironment = {
  getWorker(_: any, label: string) {
    if (label === "json") {
      return new jsonWorker();
    }
    if (label === "css" || label === "scss" || label === "less") {
      return new cssWorker();
    }
    if (label === "html" || label === "handlebars" || label === "razor") {
      return new htmlWorker();
    }
    if (label === "typescript" || label === "javascript") {
      return new tsWorker();
    }
    return new editorWorker();
  },
};
monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
  noSemanticValidation: true,
  noSuggestionDiagnostics: true,
});
monaco.languages.typescript.javascriptDefaults.addExtraLib(transpilerDecl);
monaco.languages.typescript.javascriptDefaults.addExtraLib(utilsDecl, "file:///node_modules/@types/utils/index.d.ts");
