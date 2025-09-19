import type { Component } from 'solid-js';
import { Router, Route } from '@solidjs/router';
import Dynamic from './pages/Dynamic';

const App: Component = () => {
  return (
    <Router>
      <Route path="/" component={() => <h1>Hello World</h1>} />
      <Route path="/dynamic/:name" component={Dynamic} />
    </Router>
  );
};

export default App;
