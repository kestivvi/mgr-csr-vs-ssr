import { useParams } from 'react-router-dom';

function Dynamic() {
  const { name } = useParams<{ name: string }>();

  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
    </div>
  );
}

export default Dynamic; 