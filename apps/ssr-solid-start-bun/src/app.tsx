import './global.css';
import { MetaProvider, Title } from "@solidjs/meta";
import { Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense } from "solid-js";

export default function App() {
  return (
    <MetaProvider>
      <Router root={props => (
        <Suspense fallback={<div>Loading...</div>}>
          <Title>SolidStart</Title>
          {props.children}
        </Suspense>
      )}>
        <FileRoutes />
      </Router>
    </MetaProvider>
  );
}
