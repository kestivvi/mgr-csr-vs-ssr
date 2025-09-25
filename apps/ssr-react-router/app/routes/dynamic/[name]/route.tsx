import type { Route } from "./+types/route";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: "Dynamic Page" },
    { name: "description", content: "Dynamic page with parameter" },
  ];
}

export default function DynamicPage({ params }: Route.ComponentProps) {
  const { name } = params;
  
  return (
    <div>
      <h1>Hello, {name}</h1>
    </div>
  );
}
