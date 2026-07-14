use sha2::{Digest, Sha256};
use tpt_hf_hub::{HubClient, NoopProgressReporter};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[tokio::test]
async fn downloads_a_file_and_verifies_hash() {
    let server = MockServer::start().await;
    let body = b"hello from the hub".to_vec();
    let etag = sha256_hex(&body);

    Mock::given(method("GET"))
        .and(path("/gpt2/resolve/main/config.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(body.clone())
                .insert_header("x-linked-etag", etag.as_str()),
        )
        .mount(&server)
        .await;

    let tmp_dir = tempdir();
    let client = HubClient::with_cache_dir(tmp_dir.clone()).with_endpoint(server.uri());

    let path = client
        .download_file("gpt2", "config.json", &NoopProgressReporter)
        .await
        .expect("download should succeed");

    assert!(path.is_file());
    assert!(!path.with_extension("json.tmp").exists());
    let contents = tokio::fs::read(&path).await.unwrap();
    assert_eq!(contents, body);

    // Second call should hit the cache and not require a network request.
    let path2 = client
        .download_file("gpt2", "config.json", &NoopProgressReporter)
        .await
        .expect("cached download should succeed");
    assert_eq!(path, path2);

    cleanup(tmp_dir);
}

#[tokio::test]
async fn resumes_a_partial_download_via_range_request() {
    let server = MockServer::start().await;
    let full_body = b"0123456789abcdefghij".to_vec();
    let etag = sha256_hex(&full_body);
    let already_have = &full_body[..10];
    let remainder = &full_body[10..];

    Mock::given(method("GET"))
        .and(path("/gpt2/resolve/main/model.bin"))
        .and(header("Range", "bytes=10-"))
        .respond_with(
            ResponseTemplate::new(206)
                .set_body_bytes(remainder.to_vec())
                .insert_header("x-linked-etag", etag.as_str()),
        )
        .mount(&server)
        .await;

    let tmp_dir = tempdir();
    let client = HubClient::with_cache_dir(tmp_dir.clone()).with_endpoint(server.uri());

    let dest_dir = tmp_dir.join("gpt2").join("main");
    tokio::fs::create_dir_all(&dest_dir).await.unwrap();
    tokio::fs::write(dest_dir.join("model.bin.tmp"), already_have)
        .await
        .unwrap();

    let path = client
        .download_file("gpt2", "model.bin", &NoopProgressReporter)
        .await
        .expect("resumed download should succeed");

    let contents = tokio::fs::read(&path).await.unwrap();
    assert_eq!(contents, full_body);

    cleanup(tmp_dir);
}

#[tokio::test]
async fn rejects_a_file_whose_hash_does_not_match() {
    let server = MockServer::start().await;
    let body = b"tampered content".to_vec();

    Mock::given(method("GET"))
        .and(path("/gpt2/resolve/main/config.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(body)
                .insert_header("x-linked-etag", "0".repeat(64).as_str()),
        )
        .mount(&server)
        .await;

    let tmp_dir = tempdir();
    let client = HubClient::with_cache_dir(tmp_dir.clone()).with_endpoint(server.uri());

    let result = client
        .download_file("gpt2", "config.json", &NoopProgressReporter)
        .await;

    assert!(matches!(
        result,
        Err(tpt_hf_hub::HubError::HashMismatch { .. })
    ));

    cleanup(tmp_dir);
}

#[tokio::test]
async fn snapshot_download_fetches_every_sibling_file() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/models/gpt2/revision/main"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "vocab.txt"},
            ]
        })))
        .mount(&server)
        .await;

    for name in ["config.json", "vocab.txt"] {
        Mock::given(method("GET"))
            .and(path(format!("/gpt2/resolve/main/{name}")))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(name.as_bytes().to_vec()))
            .mount(&server)
            .await;
    }

    let tmp_dir = tempdir();
    let client = HubClient::with_cache_dir(tmp_dir.clone()).with_endpoint(server.uri());

    let snapshot_dir = client
        .snapshot_download("gpt2", &NoopProgressReporter)
        .await
        .expect("snapshot download should succeed");

    assert!(snapshot_dir.join("config.json").is_file());
    assert!(snapshot_dir.join("vocab.txt").is_file());

    cleanup(tmp_dir);
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tpt-hf-hub-test-{}", uuid_like()));
    dir
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos}-{:?}", std::thread::current().id())
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn cleanup(dir: std::path::PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}
