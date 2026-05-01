import { useParams } from "@solidjs/router";

export default function HelloWorldPage() {
  const params = useParams<{ name: string }>();
  
  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {params.name}</p>
    </div>
  );
}
