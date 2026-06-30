import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_job_url(client: AsyncClient):
    response = await client.post("/jobs", data={"url": "https://ikea.com/sofa"})
    assert response.status_code == 201
    body = response.json()
    assert "id" in body
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_create_job_image(client: AsyncClient):
    response = await client.post(
        "/jobs",
        files={"image": ("sofa.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_create_job_no_input(client: AsyncClient):
    response = await client.post("/jobs")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_jobs_empty(client: AsyncClient):
    response = await client.get("/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient):
    response = await client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_after_create(client: AsyncClient):
    create = await client.post("/jobs", data={"url": "https://example.com/chair"})
    job_id = create.json()["id"]
    response = await client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["status"] == "queued"
    assert body["input_type"] == "url"
