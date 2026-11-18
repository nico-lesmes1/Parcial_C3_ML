let dataset = [];
let locationMedians = {};
let stratumMedians = {};

const currency = new Intl.NumberFormat('es-CO', {
    style: 'currency',
    currency: 'COP',
    maximumFractionDigits: 0
});

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');

    const targetName = screenId === 'home' ? 'Inicio' : screenId === 'predict' ? 'Predecir' : 'Dashboard';
    document.querySelectorAll('.nav-btn').forEach(button => {
        if (button.textContent.trim() === targetName) button.classList.add('active');
    });
}

function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const headers = lines.shift().split(',');
    return lines.map(line => {
        const values = line.split(',');
        if (values.length !== headers.length) return null;
        const row = {};
        headers.forEach((header, i) => {
            const value = values[i];
            row[header] = ['precio', 'area_construida', 'area_total', 'area_privada', 'estrato',
                'habitaciones', 'baños', 'parqueaderos', 'administracion'].includes(header)
                ? Number(value) || 0
                : value;
        });
        return row;
    }).filter(Boolean);
}

function median(values) {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

async function loadDataset() {
    try {
        const response = await fetch('../dataset_limpio.csv');
        if (!response.ok) throw new Error(`No se pudo cargar el dataset (${response.status})`);
        dataset = parseCSV(await response.text());

        const locations = [...new Set(dataset.map(row => row.ubicacion_norm).filter(Boolean))].sort();
        const select = document.getElementById('ubicacion');
        select.innerHTML = locations.map(location =>
            `<option value="${location}">${location.replace(/\b\w/g, char => char.toUpperCase())}</option>`
        ).join('');
        select.value = locations.includes('cedritos') ? 'cedritos' : locations[0];

        locationMedians = {};
        locations.forEach(location => {
            locationMedians[location] = median(dataset
                .filter(row => row.ubicacion_norm === location)
                .map(row => row.precio));
        });

        stratumMedians = {};
        for (let estrato = 1; estrato <= 6; estrato++) {
            stratumMedians[estrato] = median(dataset
                .filter(row => row.estrato === estrato)
                .map(row => row.precio));
        }

        renderClusterChart();
    } catch (error) {
        console.error(error);
        document.getElementById('ubicacion').innerHTML = '<option value="">Dataset no disponible</option>';
    }
}

function predictSupervised(property) {
    if (!dataset.length) {
        const factor = 1 + (property.estrato - 1) * 0.23;
        return property.area * 2800000 * factor + property.parqueaderos * 35000000;
    }

    const candidates = dataset.filter(row =>
        row.area_construida > 0 &&
        row.precio > 0 &&
        row.estrato > 0
    );
    const scale = {
        area: 180,
        estrato: 5,
        rooms: 8,
        baths: 6,
        parking: 5,
        admin: 5000000
    };

    const scored = candidates.map(row => {
        const distance =
            Math.pow((property.area - row.area_construida) / scale.area, 2) +
            Math.pow((property.estrato - row.estrato) / scale.estrato, 2) +
            Math.pow((property.habitaciones - (row.habitaciones || 0)) / scale.rooms, 2) +
            Math.pow((property.banos - (row.baños || 0)) / scale.baths, 2) +
            Math.pow((property.parqueaderos - (row.parqueaderos || 0)) / scale.parking, 2) +
            Math.pow((property.admin - (row.administracion || 0)) / scale.admin, 2);
        return { row, distance };
    }).sort((a, b) => a.distance - b.distance).slice(0, 40);

    let weightedSum = 0;
    let weightSum = 0;
    scored.forEach(({ row, distance }) => {
        const weight = 1 / (0.08 + distance);
        weightedSum += row.precio * weight;
        weightSum += weight;
    });

    const locationMedian = locationMedians[property.location] || median(candidates.map(row => row.precio));
    const stratumMedian = stratumMedians[property.estrato] || median(candidates.map(row => row.precio));
    const comparableMedian = weightedSum / weightSum;
    return Math.round((comparableMedian * 0.65 + locationMedian * 0.25 + stratumMedian * 0.10) / 100000) * 100000;
}

function kmeansProperties() {
    if (!dataset.length) return null;
    const rows = dataset.slice().sort(() => 0.5 - Math.random()).slice(0, 3500);
    const features = rows.map(row => [
        row.area_construida,
        row.estrato,
        row.precio / 1000000
    ]);

    const mins = [0, 1, 0];
    const maxs = [970, 6, 17000];
    const normalized = features.map(point => point.map((value, i) =>
        (value - mins[i]) / (maxs[i] - mins[i])
    ));

    const step = Math.max(1, Math.floor(normalized.length / 3));
    let centroids = [normalized[0], normalized[step], normalized[step * 2]];

    for (let iteration = 0; iteration < 25; iteration++) {
        const labels = normalized.map(point => {
            let best = 0;
            let bestDistance = Infinity;
            centroids.forEach((centroid, index) => {
                const distance = point.reduce((sum, value, i) =>
                    sum + Math.pow(value - centroid[i], 2), 0);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = index;
                }
            });
            return best;
        });

        const next = centroids.map((_, cluster) => {
            const points = normalized.filter((_, i) => labels[i] === cluster);
            if (!points.length) return centroids[cluster];
            return mins.map((_, i) =>
                points.reduce((sum, point) => sum + point[i], 0) / points.length
            );
        });
        if (JSON.stringify(next) === JSON.stringify(centroids)) break;
        centroids = next;
    }

    const labels = normalized.map(point => {
        let best = 0;
        let bestDistance = Infinity;
        centroids.forEach((centroid, index) => {
            const distance = point.reduce((sum, value, i) =>
                sum + Math.pow(value - centroid[i], 2), 0);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = index;
            }
        });
        return best;
    });

    return centroids.map((centroid, index) => {
        const area = centroid[0] * (maxs[0] - mins[0]) + mins[0];
        const estrato = centroid[1] * (maxs[1] - mins[1]) + mins[1];
        const price = (centroid[2] * (maxs[2] - mins[2]) + mins[2]) * 1000000;
        const count = labels.filter(label => label === index).length;
        return { area, estrato, price, count };
    }).sort((a, b) => a.price - b.price);
}

function assignCluster(property) {
    if (property.area > 200 && property.estrato >= 5) {
        return { name: 'Premium', code: 'Cluster 1', index: 2 };
    }
    if (property.area < 100 && property.estrato <= 4) {
        return { name: 'Económico', code: 'Cluster 0', index: 0 };
    }
    return { name: 'Medio', code: 'Cluster 2', index: 1 };
}

function setBadge(type, text) {
    const badge = document.getElementById('result-badge');
    badge.textContent = text;
    const styles = {
        green: ['rgba(16,185,129,.15)', '#10b981', 'rgba(16,185,129,.3)'],
        gold: ['rgba(245,158,11,.15)', '#f59e0b', 'rgba(245,158,11,.3)'],
        red: ['rgba(239,68,68,.15)', '#ef4444', 'rgba(239,68,68,.3)']
    };
    const [background, color, border] = styles[type];
    badge.style.background = background;
    badge.style.color = color;
    badge.style.borderColor = border;
}

function calculatePrice() {
    const property = {
        area: Number(document.getElementById('area').value),
        habitaciones: Number(document.getElementById('habitaciones').value),
        banos: Number(document.getElementById('banos').value),
        estrato: Number(document.getElementById('estrato').value),
        parqueaderos: Number(document.getElementById('parqueaderos').value),
        admin: Number(document.getElementById('admin').value),
        location: document.getElementById('ubicacion').value,
        publishedPrice: Number(document.getElementById('precio-publicado').value) || null
    };

    const predicted = predictSupervised(property);
    const cluster = assignCluster(property);
    const realPrice = property.publishedPrice;
    const gap = realPrice ? (realPrice - predicted) / predicted : null;

    let position = 'Justo';
    if (gap !== null) {
        if (gap < -0.15) position = 'Oportunidad';
        else if (gap > 0.15) position = 'Sobrevalorado';
    }

    document.getElementById('result-price').textContent = currency.format(predicted);
    document.getElementById('result-cluster').textContent = `${cluster.name} (${cluster.code})`;
    document.getElementById('result-m2').textContent = currency.format(predicted / property.area);
    document.getElementById('result-subtitle').textContent =
        realPrice ? `Precio publicado: ${currency.format(realPrice)} · Brecha: ${(gap * 100).toFixed(1)}%` :
        'Estimación supervisada con comparables reales del dataset';

    if (position === 'Oportunidad') setBadge('green', '🟢 Oportunidad de inversión');
    else if (position === 'Sobrevalorado') setBadge('red', '🔴 Sobrevalorada');
    else setBadge('gold', '🟡 Precio justo / por negociar');

    const policy = getRLPolicy(cluster.index, position);
    document.getElementById('result-agent').textContent = policy.action;
    document.getElementById('result-agent-detail').textContent =
        `${policy.reason} Estado MDP: ${['Económico', 'Medio', 'Premium'][cluster.index]} × ${position}.`;

    showScreen('result');
    renderPriceChart(predicted, realPrice, cluster);
}

function getRLPolicy(segment, position) {
    if (position === 'Oportunidad') {
        return {
            action: 'COMPRAR',
            reason: 'El valor estimado supera el precio publicado y la recompensa esperada favorece cerrar.'
        };
    }
    if (position === 'Sobrevalorado') {
        return segment === 0
            ? { action: 'SEGUIR BUSCANDO', reason: 'El ahorro no compensa cerrar ahora; conviene esperar otro anuncio.' }
            : { action: 'NEGOCIAR FUERTE', reason: 'Existe sobreprecio; una oferta cercana a -15 % maximiza el ahorro esperado.' };
    }
    if (segment === 2) {
        return { action: 'NEGOCIAR FUERTE', reason: 'En premium, negociar reduce el costo sin abandonar la propiedad.' };
    }
    return { action: 'NEGOCIAR FUERTE', reason: 'El precio está cerca del valor justo; una negociación moderada-fuerte mejora el retorno.' };
}

let priceChartInstance = null;

function renderPriceChart(predicted, realPrice, cluster) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    if (priceChartInstance) priceChartInstance.destroy();

    const averages = {
        'Económico': 387000000,
        'Medio': 529000000,
        'Premium': 1641000000
    };
    const labels = Object.keys(averages);
    const values = Object.values(averages);
    labels.push('Tu propiedad');
    values.push(realPrice || predicted);

    priceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: ['#94a3b8', '#6366f1', '#f59e0b', '#10b981'].map((color, i) =>
                    i === 3 ? 'rgba(16,185,129,.8)' : color + '99'),
                borderColor: ['#94a3b8', '#6366f1', '#f59e0b', '#10b981'],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: context => currency.format(context.raw) } }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8',
                        callback: value => '$' + (value / 1000000) + 'M'
                    },
                    grid: { color: 'rgba(255,255,255,.05)' }
                },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            }
        }
    });
}

function renderClusterChart() {
    const ctx = document.getElementById('clusterChart');
    if (!ctx) return;
    const clusters = kmeansProperties();
    const fallback = [
        { area: 96, estrato: 3.4, price: 387000000, count: 2800 },
        { area: 89, estrato: 4.4, price: 529000000, count: 3000 },
        { area: 267, estrato: 5.3, price: 1641000000, count: 2572 }
    ];
    const data = (clusters || fallback).map((cluster, index) => ({
        label: ['Económico', 'Medio', 'Premium'][index],
        data: [{ x: cluster.area, y: cluster.estrato, r: Math.max(9, Math.sqrt(cluster.count || 250) / 1.8) }],
        backgroundColor: ['rgba(148,163,184,.65)', 'rgba(99,102,241,.65)', 'rgba(245,158,11,.65)'][index]
    }));

    new Chart(ctx, {
        type: 'bubble',
        data: { datasets: data },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#f8fafc' } },
                tooltip: {
                    callbacks: {
                        label: context => `Área: ${context.raw.x.toFixed(0)} m² | Estrato: ${context.raw.y.toFixed(1)}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Área construida (m²)', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,.05)' }
                },
                y: {
                    title: { display: true, text: 'Estrato', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,.05)' }
                }
            }
        }
    });
}

window.addEventListener('DOMContentLoaded', loadDataset);
