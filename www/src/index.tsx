import React from "react";
import { createRoot } from "react-dom/client";
import iFrameResize from "iframe-resizer/js/iframeResizer";

import { App } from "./app";

const app = document.getElementById("app") as HTMLElement;
const root = createRoot(app);
root.render(<App />);

iFrameResize({}, ".demo-iframe");