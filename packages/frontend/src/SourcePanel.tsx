import axios from "axios";
import { useRef, useState } from "react";
import MonacoEditor from "react-monaco-editor";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import { HOST } from "./config";
import { useRule } from "./MonacoRule";
import { transpileResult$, useTranspileResult } from "./MonacoRule";
import { jid$ } from "./ResultPanel";

const DEFAULT_SRC = `<!DOCTYPE html>
<title>Hello, World</title>
<button id="hello" onclick="addContent()">Click me</button>
<script>
  function addContent() {
    document.body.append("Hello, JavaScript");
  }
</script>
`;

export default function SourcePanel() {
  const [editor, setEditor] = useState(false);
  const [editorValue, setEditorValue] = useState(DEFAULT_SRC);
  const fileInput = useRef<HTMLInputElement>(null);

  const rule = useRule();
  const transpileResult = useTranspileResult();

  function getLang() {
    if (!transpileResult.success) {
      return "plaintext";
    }
    switch (transpileResult.category) {
      case "web":
        return "html";
      case "graphics.turtle":
        return "python";
      default:
        return "plaintext";
    }
  }

  async function submit() {
    if (!transpileResult.success) {
      alert("规则尚有错误");
      return;
    }
    let file: Blob;
    if (editor) {
      let mimeType: string;
      switch (transpileResult.category) {
        case "web":
          mimeType = "text/html";
          break;
        default:
          mimeType = "text/plain";
          break;
      }
      file = new Blob([editorValue], { type: mimeType });
    } else {
      if (!fileInput.current?.files?.length) {
        alert("没有文件");
        return;
      }
      file = fileInput.current.files[0];
    }
    const data = new FormData();
    data.set("rule", rule);
    data.set("category", transpileResult.category ?? "");
    data.set("file", file);
    const result = await axios.post(`${HOST}/judge`, data);
    if (result.data.success) {
      jid$.next(result.data.id);
    }
  }

  return (
    <div className="h-full w-full flex flex-col">
      <div className="p-2 flex justify-between bg-base-200">
        <span className="flex gap-4">
          <label className="flex gap-1 items-center">
            <input
              type="radio"
              name="sourceType"
              value="file"
              className="radio radio-primary radio-sm"
              checked={!editor}
              onChange={(e) => setEditor(!e.target.checked)}
            ></input>
            <span>上传文件</span>
          </label>
          <label className="flex gap-1 items-center">
            <input
              type="radio"
              name="sourceType"
              value="editor"
              className="radio radio-primary radio-sm"
              checked={editor}
              onChange={(e) => setEditor(e.target.checked)}
            ></input>
            <span>编辑器</span>
          </label>
        </span>
        <button
          className="btn btn-primary btn-sm"
          disabled={!transpileResult.success}
          onClick={submit}
        >
          {transpileResult.success ? "提交" : "规则有误"}
        </button>
      </div>
      <div>
        当前类别{" "}
        {transpileResult.success
          ? transpileResult.category
          : transpileResult.message}
      </div>
      <div className="flex-grow">
        {editor ? (
          <MonacoEditor
            language={getLang()}
            value={editorValue}
            onChange={setEditorValue}
            options={{
              automaticLayout: true,
            }}
          ></MonacoEditor>
        ) : (
          <div className="p-2">
            <input
              type="file"
              className="file-input file-input-bordered w-full"
              ref={fileInput}
            ></input>
          </div>
        )}
      </div>
    </div>
  );
}
