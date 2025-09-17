<script lang="ts">
	import { page } from '$app/stores';
	
	$: name = $page.params.name;
</script>

<div>
	<h1>Hello, {name}</h1>
</div>
