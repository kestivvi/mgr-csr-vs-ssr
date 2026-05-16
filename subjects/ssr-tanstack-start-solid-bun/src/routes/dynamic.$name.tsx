import { createSignal } from 'solid-js';
import { createFileRoute } from '@tanstack/solid-router';

export const Route = createFileRoute('/dynamic/$name')({
  component: Dynamic,
});

function Dynamic() {
  const params = Route.useParams();
  const [count, setCount] = createSignal(0);

  return (
    <main>
      <h1>Hello World</h1>
      <p>Dynamic ID: {params.name}</p>
      <div>
        <p>Count: {count()}</p>
        <button onClick={() => setCount(count() + 1)}>Increment</button>
      </div>
    </main>
  );
}
