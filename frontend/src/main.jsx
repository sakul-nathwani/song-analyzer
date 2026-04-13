import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ClerkProvider } from "@clerk/clerk-react";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <ClerkProvider publishableKey="pk_test_bXV0dWFsLWNvYnJhLTk0LmNsZXJrLmFjY291bnRzLmRldiQ">
      <App />
    </ClerkProvider>
  </StrictMode>
);
