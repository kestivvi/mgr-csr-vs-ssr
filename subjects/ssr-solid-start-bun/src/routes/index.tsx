import { createSignal } from 'solid-js';

export default function Home() {
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
}
