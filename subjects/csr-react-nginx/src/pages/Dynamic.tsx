import { useState } from 'react';
import { useParams } from 'react-router-dom';

function Dynamic() {
  const { name } = useParams<{ name: string }>();
  const [count, setCount] = useState(0);

  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
      <div>
        <p>Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </div>
    </div>
  );
}

export default Dynamic;