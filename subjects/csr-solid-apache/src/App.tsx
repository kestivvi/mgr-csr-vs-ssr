import { createSignal, type Component } from 'solid-js';
import { Router, Route } from '@solidjs/router';
import Dynamic from './pages/Dynamic';

const Home: Component = () => {
  const [count, setCount] = createSignal(0);
  return (
    <main>
      <h1>Hello World</h1>
      <div>
        <p>Count: {count()}</p>
        <button onClick={() => setCount(count() + 1)}>Increment</button>
      </div>
    </main>
  );
};

const App: Component = () => {
  return (
    <Router>
      <Route path="/" component={Home} />
      <Route path="/dynamic/:name" component={Dynamic} />
    </Router>
  );
};

export default App;
