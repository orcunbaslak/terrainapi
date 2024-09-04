use actix_multipart::Multipart;
use actix_web::{web, App, Error, HttpResponse, HttpServer};
use csv::ReaderBuilder;
use futures::{StreamExt, TryStreamExt};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use kiddo::KdTree;

#[derive(Deserialize)]
struct SmoothingRequest {
    smoothing_factor: f64,
}

#[derive(Serialize)]
struct TerrainPoint {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Deserialize)]
struct ElevationQuery {
    x: f64,
    y: f64,
}

async fn query_elevation(
    query: web::Query<ElevationQuery>,
    data: web::Data<TerrainData>,
) -> Result<HttpResponse, Error> {
    let elevation = data.interpolate_elevation(query.x, query.y);
    Ok(HttpResponse::Ok().json(serde_json::json!({ "elevation": elevation })))
}

struct TerrainData {
    kdtree: KdTree<f64, u32, 2>,
    grid: Array2<f64>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl TerrainData {
    fn new(points: Vec<(f64, f64, f64)>, grid: Array2<f64>, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        let mut kdtree: KdTree<f64, u32, 2> = KdTree::new();
        for (i, &(x, y, _)) in points.iter().enumerate() {
            kdtree.add(&[x, y], i as u32);
        }
        Self { kdtree, grid, min_x, max_x, min_y, max_y }
    }

    fn interpolate_elevation(&self, x: f64, y: f64) -> f64 {
        let x_norm = (x - self.min_x) / (self.max_x - self.min_x);
        let y_norm = (y - self.min_y) / (self.max_y - self.min_y);
        
        let (rows, cols) = self.grid.dim();
        let x_grid = x_norm * (cols - 1) as f64;
        let y_grid = y_norm * (rows - 1) as f64;
        
        let x0 = x_grid.floor() as usize;
        let x1 = (x0 + 1).min(cols - 1);
        let y0 = y_grid.floor() as usize;
        let y1 = (y0 + 1).min(rows - 1);
        
        let x_frac = x_grid - x0 as f64;
        let y_frac = y_grid - y0 as f64;
        
        let z00 = self.grid[[y0, x0]];
        let z01 = self.grid[[y0, x1]];
        let z10 = self.grid[[y1, x0]];
        let z11 = self.grid[[y1, x1]];
        
        let z0 = z00 * (1.0 - x_frac) + z01 * x_frac;
        let z1 = z10 * (1.0 - x_frac) + z11 * x_frac;
        
        z0 * (1.0 - y_frac) + z1 * y_frac
    }
}

async fn smooth_terrain(mut payload: Multipart, query: web::Query<SmoothingRequest>) -> Result<HttpResponse, Error> {
    let mut csv_data = Vec::with_capacity(1024 * 1024);
    
    while let Ok(Some(mut field)) = payload.try_next().await {
        while let Some(chunk) = field.next().await {
            let data = chunk?;
            csv_data.extend_from_slice(&data);
        }
    }

    let cursor = Cursor::new(csv_data);
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(cursor);
    
    let mut points = Vec::new();
    for record in csv_reader.records() {
        let record = record.map_err(|e| actix_web::error::ErrorBadRequest(format!("Failed to read CSV record: {}", e)))?;
        let x: f64 = record[0].parse().map_err(|e| actix_web::error::ErrorBadRequest(format!("Failed to parse X: {}", e)))?;
        let y: f64 = record[1].parse().map_err(|e| actix_web::error::ErrorBadRequest(format!("Failed to parse Y: {}", e)))?;
        let z: f64 = record[2].parse().map_err(|e| actix_web::error::ErrorBadRequest(format!("Failed to parse Z: {}", e)))?;
        points.push((x, y, z));
    }

    let resolution = (points.len() as f64).sqrt().ceil() as usize;

    let (min_x, max_x, min_y, max_y) = points.iter()
        .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN),
              |(min_x, max_x, min_y, max_y), &(x, y, _)| {
                  (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
              });
    let mut grid = Array2::<f64>::zeros((resolution, resolution));
    let mut count = Array2::<f64>::zeros((resolution, resolution));

    for &(x, y, z) in &points {
        let i = ((x - min_x) / (max_x - min_x) * (resolution as f64 - 1.0)).round() as usize;
        let j = ((y - min_y) / (max_y - min_y) * (resolution as f64 - 1.0)).round() as usize;
        grid[[i, j]] += z;
        count[[i, j]] += 1.0;
    }

    for ((i, j), value) in grid.indexed_iter_mut() {
        if let Some(count_value) = count.get([i, j]) {
            if *count_value > 0.0 {
                *value /= *count_value;
            }
        }
    }

    let kernel_size = (query.smoothing_factor * 10.0).round() as usize;
    let smoothed_grid = if kernel_size > 0 {
        let kernel = create_gaussian_kernel(kernel_size, query.smoothing_factor);
        convolve2d(&grid, &kernel)
    } else {
        grid
    };

    let smoothed_points: Vec<TerrainPoint> = smoothed_grid.indexed_iter()
        .map(|((i, j), &z)| {
            let x = min_x + (i as f64 / (resolution - 1) as f64) * (max_x - min_x);
            let y = min_y + (j as f64 / (resolution - 1) as f64) * (max_y - min_y);
            TerrainPoint { x, y, z }
        })
        .collect();

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "points": smoothed_points,
        "terrain_data": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }
    })))
}

fn create_gaussian_kernel(size: usize, sigma: f64) -> Array2<f64> {
    let mut kernel = Array2::zeros((size, size));
    let center = size as f64 / 2.0;
    let mut sum = 0.0;

    for i in 0..size {
        for j in 0..size {
            let x = i as f64 - center + 0.5;
            let y = j as f64 - center + 0.5;
            let value = (-((x * x + y * y) / (2.0 * sigma * sigma))).exp();
            kernel[[i, j]] = value;
            sum += value;
        }
    }

    kernel.mapv_inplace(|v| v / sum);
    kernel
}

fn convolve2d(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (m, n) = input.dim();
    let (km, kn) = kernel.dim();
    let mut output = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for ki in 0..km {
                for kj in 0..kn {
                    let ii = (i as isize + ki as isize - km as isize / 2).rem_euclid(m as isize) as usize;
                    let jj = (j as isize + kj as isize - kn as isize / 2).rem_euclid(n as isize) as usize;
                    let weight = kernel[[ki, kj]];
                    sum += input[[ii, jj]] * weight;
                    weight_sum += weight;
                }
            }
            output[[i, j]] = if weight_sum > 0.0 { sum / weight_sum } else { input[[i, j]] };
        }
    }

    output
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting server");

    let default_terrain = web::Data::new(TerrainData::new(
        Vec::new(),
        Array2::zeros((1, 1)),
        0.0,
        1.0,
        0.0,
        1.0,
    ));

    HttpServer::new(move || {
        App::new()
            .app_data(default_terrain.clone())
            .service(web::resource("/smooth").route(web::post().to(smooth_terrain)))
            .service(web::resource("/elevation").route(web::get().to(query_elevation)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
