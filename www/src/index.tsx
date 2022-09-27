
import "bootstrap/dist/css/bootstrap.min.css";
import React from "react";
import { createRoot } from "react-dom/client";

import { App } from "./app";

const app = document.getElementById("app") as HTMLElement;
const root = createRoot(app);
root.render(<App />);