<script lang="ts">
	import { page } from '$app/stores';
	
	$: name = $page.params.name;
</script>

<div>
	<h1>Hello World</h1>
	<p>Dynamic ID: {name}</p>
</div>
